//
//  BusInfoView.swift
//  Buber
//
//  Created by Alexey Salangin on 11/16/19.
//  Copyright © 2019 Alexey Salangin. All rights reserved.
//

import UIKit

final class BusInfoView: UIView {
    private let contentStackView = UIStackView()

    init() {
        super.init(frame: .zero)
        self.setupSubviews()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupSubviews() {
        self.addSubview(self.contentStackView)
        self.contentStackView.snp.makeConstraints { make in
            make.edges.equalToSuperview()
        }

        self.contentStackView.axis = .vertical
        self.contentStackView.spacing = 8

        let numberLabel = UILabel()
        numberLabel.font = UIFont.appBold(21)
        numberLabel.text = "NO. 550"
        numberLabel.snp.makeConstraints { make in
            make.height.equalTo(25)
        }

        let modelLabel = UILabel()
        modelLabel.font = UIFont.appBold(16)
        modelLabel.text = "Elfvik Quarry"
        modelLabel.snp.makeConstraints { make in
            make.height.equalTo(19)
        }

        let busView = UIStackView()
        busView.axis = .horizontal
        busView.spacing = 15

        busView.snp.makeConstraints { make in
            make.height.equalTo(39)
        }

        let busImageView = UIImageView(image: UIImage(named: "itakeskus"))
        busImageView.snp.makeConstraints { make in
            make.width.equalTo(70)
        }

        busView.addArrangedSubview(busImageView)

        let costLabel = UILabel()
        costLabel.font = UIFont.appMedium(16)
        costLabel.text = "4,8 €"

        let durationLabel = UILabel()
        durationLabel.font = UIFont.appRegular(12)
        durationLabel.text = "4 mins"

        let busDataView = UIStackView(arrangedSubviews: [costLabel, durationLabel])
        busDataView.axis = .vertical
        busDataView.distribution = .fillEqually
        busView.addArrangedSubview(busDataView)

        let driverLabel = UILabel()
        driverLabel.font = UIFont.appMedium(16)
        driverLabel.text = "Driver"
        driverLabel.snp.makeConstraints { make in
            make.height.equalTo(19)
        }

        let driverNameLabel = UILabel()
        driverNameLabel.font = UIFont.appRegular(16)
        driverNameLabel.text = "Jacob M."
        driverNameLabel.snp.makeConstraints { make in
            make.height.equalTo(21)
        }

        let busViewWrapper = UIView()
        busViewWrapper.addSubview(busView)
        busView.snp.makeConstraints { make in
            make.edges.equalToSuperview().inset(UIEdgeInsets(top: 18, left: 0, bottom: 12, right: 0))
        }

        self.contentStackView.addArrangedSubview(numberLabel)
        self.contentStackView.addArrangedSubview(modelLabel)
        self.contentStackView.addArrangedSubview(busViewWrapper)
        self.contentStackView.addArrangedSubview(driverLabel)
        self.contentStackView.addArrangedSubview(driverNameLabel)

        self.snp.makeConstraints { make in
            make.height.equalTo(185)
        }
    }
}
